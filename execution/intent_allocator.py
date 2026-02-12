from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any, Dict

_INIT_LOCK = threading.Lock()


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


def _safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return int(default)


def _cfg(bot: Any, name: str, default: Any) -> Any:
    try:
        return getattr(getattr(bot, "cfg", None), name, default)
    except Exception:
        return default


def _sanitize_token(value: Any, *, fallback: str = "NA", max_len: int = 24) -> str:
    s = str(value or "").strip()
    if not s:
        s = fallback
    out_chars: list[str] = []
    for ch in s:
        if ch.isalnum() or ch in ("_", "-"):
            out_chars.append(ch)
        elif ch in (".", "/", " ", ":", "|"):
            out_chars.append("_")
    out = "".join(out_chars).strip("_")
    if not out:
        out = fallback
    return out[: max(1, int(max_len))]


def _symkey(sym: str) -> str:
    s = (sym or "").upper().strip()
    s = s.replace("/USDT:USDT", "USDT").replace("/USDT", "USDT")
    s = s.replace(":USDT", "USDT").replace(":", "")
    s = s.replace("/", "")
    if s.endswith("USDTUSDT"):
        s = s[:-4]
    return s


def _state_path(bot: Any) -> Path:
    p = str(_cfg(bot, "INTENT_ALLOCATOR_STATE_PATH", "") or "").strip()
    if not p:
        p = str(os.getenv("INTENT_ALLOCATOR_STATE_PATH", "") or "").strip()
    if not p:
        p = "logs/intent_allocator_state.json"
    return Path(p)


def _enabled(bot: Any) -> bool:
    raw = _cfg(bot, "INTENT_ALLOCATOR_ENABLED", None)
    if raw is not None:
        return _truthy(raw)
    return _truthy(os.getenv("INTENT_ALLOCATOR_ENABLED", "1"))


def _ensure_run_context(bot: Any) -> Dict[str, Any]:
    st = getattr(bot, "state", None)
    if st is None:
        return {}
    rc = getattr(st, "run_context", None)
    if isinstance(rc, dict):
        return rc
    try:
        st.run_context = {}
        rc = st.run_context
    except Exception:
        rc = {}
    return rc if isinstance(rc, dict) else {}


def _persist(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tid = str(threading.get_ident())
    tmp = path.with_suffix(path.suffix + f".{tid}.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=True, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def _load_state(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        raw = path.read_text(encoding="utf-8")
        data = json.loads(raw)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _ensure_store(bot: Any) -> Dict[str, Any]:
    if not _enabled(bot):
        return {}
    with _INIT_LOCK:
        rc = _ensure_run_context(bot)
        store = rc.get("intent_allocator")
        if isinstance(store, dict) and store.get("loaded"):
            return store

        path = _state_path(bot)
        state = _load_state(path)
        bot_instance = _sanitize_token(
            _cfg(bot, "BOT_INSTANCE_ID", "") or os.getenv("BOT_INSTANCE_ID", ""),
            fallback="BOT",
            max_len=16,
        )
        run_epoch_cfg = _safe_int(_cfg(bot, "RUN_EPOCH", 0), 0)
        run_epoch_env = _safe_int(os.getenv("RUN_EPOCH", "0"), 0)
        run_epoch_saved = _safe_int(state.get("run_epoch"), 0)
        run_epoch = run_epoch_saved or run_epoch_cfg or run_epoch_env or int(time.time())
        seq = max(0, _safe_int(state.get("next_seq"), 1) - 1)

        store = {
            "loaded": True,
            "path": str(path),
            "bot_instance": bot_instance,
            "run_epoch": int(run_epoch),
            "seq": int(seq),
            "lock": threading.Lock(),
        }
        rc["intent_allocator"] = store
        if not state:
            _persist(path, {"bot_instance": bot_instance, "run_epoch": int(run_epoch), "next_seq": int(seq + 1)})
        return store


def _sanitize_client_order_id(raw: str, *, max_len: int = 35) -> str:
    safe = _sanitize_token(raw, fallback="SE", max_len=max_len * 2)
    if len(safe) <= max_len:
        return safe
    import hashlib

    digest = hashlib.sha1(safe.encode("utf-8")).hexdigest()[:10]
    keep = max(1, max_len - 11)
    return f"{safe[:keep]}_{digest}"[:max_len]


def allocate_intent_id(
    bot: Any,
    *,
    component: str,
    intent_kind: str,
    symbol: str,
    is_exit: bool = False,
) -> str:
    """
    Allocate a collision-resistant intent id with durable monotonic sequence.
    Format:
      I2.<bot_instance>.<run_epoch>.<component>.<intent_kind>.<symbol>.<seq>
    """
    store = _ensure_store(bot)
    if not store:
        return ""
    lock = store.get("lock")
    if not hasattr(lock, "acquire") or not hasattr(lock, "release"):
        lock = threading.Lock()
        store["lock"] = lock
    with lock:
        seq = int(store.get("seq", 0)) + 1
        store["seq"] = seq
        comp = _sanitize_token(component, fallback="component", max_len=24)
        kind = _sanitize_token(intent_kind, fallback=("EXIT" if is_exit else "ENTRY"), max_len=16).upper()
        sym = _sanitize_token(_symkey(symbol), fallback="UNK", max_len=12).upper()
        intent_id = (
            f"I2.{store.get('bot_instance','BOT')}.{int(store.get('run_epoch', int(time.time())))}."
            f"{comp}.{kind}.{sym}.{seq:08d}"
        )
        _persist(
            Path(str(store.get("path") or _state_path(bot))),
            {
                "bot_instance": str(store.get("bot_instance") or "BOT"),
                "run_epoch": int(store.get("run_epoch") or int(time.time())),
                "next_seq": int(seq + 1),
            },
        )
        return intent_id


def derive_client_order_id(
    *,
    intent_id: str,
    prefix: str,
    symbol: str,
    max_len: int = 35,
) -> str:
    """
    Deterministic exchange-safe client id derived from intent identity.
    """
    import hashlib

    pref = _sanitize_token(prefix, fallback="SE", max_len=8).upper()
    sym = _sanitize_token(_symkey(symbol), fallback="SYM", max_len=6).upper()
    blob = f"{intent_id}|{pref}|{sym}"
    digest = hashlib.sha1(blob.encode("utf-8")).hexdigest()[:16]
    out = f"{pref}_{sym}_{digest}"
    return _sanitize_client_order_id(out, max_len=max_len)
