from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

try:
    from execution import event_journal as _event_journal  # type: ignore
except Exception:
    _event_journal = None


_REUSABLE_STAGES = {"ACKED", "OPEN", "PARTIAL", "FILLED", "DONE"}
_UNKNOWN_STAGES = {"SUBMITTED_UNKNOWN", "CANCEL_SENT_UNKNOWN", "REPLACE_RACE", "OPEN_UNKNOWN"}


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
        if v != v:
            return default
        return v
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


def _ensure_run_context(bot) -> dict:
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


def _enabled(bot) -> bool:
    try:
        cfg = getattr(bot, "cfg", None)
        raw = getattr(cfg, "INTENT_LEDGER_ENABLED", None)
        if raw is not None:
            return _truthy(raw)
    except Exception:
        pass
    return _truthy(os.getenv("INTENT_LEDGER_ENABLED", "1"))


def _path_from_bot(bot) -> Path:
    p = ""
    try:
        cfg = getattr(bot, "cfg", None)
        p = str(getattr(cfg, "INTENT_LEDGER_PATH", "") or "").strip()
    except Exception:
        p = ""
    if not p:
        p = str(os.getenv("INTENT_LEDGER_PATH", "") or "").strip()
    if not p:
        p = "logs/intent_ledger.jsonl"
    return Path(p)


def _journal_path_from_bot(bot) -> Path:
    p = ""
    try:
        cfg = getattr(bot, "cfg", None)
        p = str(getattr(cfg, "EVENT_JOURNAL_PATH", "") or "").strip()
    except Exception:
        p = ""
    if not p:
        p = str(os.getenv("EVENT_JOURNAL_PATH", "") or "").strip()
    if not p:
        p = "logs/execution_journal.jsonl"
    return Path(p)


def _reuse_max_age_sec(bot) -> float:
    try:
        cfg = getattr(bot, "cfg", None)
        raw = getattr(cfg, "INTENT_LEDGER_REUSE_MAX_AGE_SEC", None)
        if raw is not None:
            return max(0.0, _safe_float(raw, 900.0))
    except Exception:
        pass
    return max(0.0, _safe_float(os.getenv("INTENT_LEDGER_REUSE_MAX_AGE_SEC", "900"), 900.0))


def _unknown_max_age_sec(bot) -> float:
    try:
        cfg = getattr(bot, "cfg", None)
        raw = getattr(cfg, "INTENT_LEDGER_UNKNOWN_MAX_AGE_SEC", None)
        if raw is not None:
            return max(0.0, _safe_float(raw, 300.0))
    except Exception:
        pass
    return max(0.0, _safe_float(os.getenv("INTENT_LEDGER_UNKNOWN_MAX_AGE_SEC", "300"), 300.0))


def _new_store(path: str, journal_path: str = "") -> dict:
    return {
        "loaded": False,
        "path": str(path or ""),
        "journal_path": str(journal_path or ""),
        "intents": {},
        "by_order_id": {},
        "by_client_order_id": {},
    }


def _ensure_store(bot) -> Optional[dict]:
    if not _enabled(bot):
        return None
    rc = _ensure_run_context(bot)
    store = rc.get("intent_ledger")
    if not isinstance(store, dict):
        store = _new_store(str(_path_from_bot(bot)), str(_journal_path_from_bot(bot)))
        rc["intent_ledger"] = store
    else:
        if not str(store.get("path") or "").strip():
            store["path"] = str(_path_from_bot(bot))
        if not str(store.get("journal_path") or "").strip():
            store["journal_path"] = str(_journal_path_from_bot(bot))
    if not bool(store.get("loaded", False)):
        _load_from_disk(store)
    return store


def _load_from_disk(store: dict) -> None:
    path = Path(str(store.get("path") or "logs/intent_ledger.jsonl"))
    store["loaded"] = True
    loaded = False
    if path.exists():
        try:
            rows: list[dict] = []
            for raw in path.read_text(encoding="utf-8").splitlines():
                line = str(raw or "").strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except Exception:
                    continue
                if isinstance(payload, dict):
                    rows.append(payload)
            rows.sort(key=lambda d: _safe_float(d.get("ts"), 0.0))
            for payload in rows:
                _apply(store, payload)
                loaded = True
        except Exception:
            return
    if loaded:
        return
    # Fallback: recover ledger state from journal mirror events.
    jpath = Path(str(store.get("journal_path") or "logs/execution_journal.jsonl"))
    if not jpath.exists():
        return
    try:
        rows: list[dict] = []
        for raw in jpath.read_text(encoding="utf-8").splitlines():
            line = str(raw or "").strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except Exception:
                continue
            if str(payload.get("event") or "").strip() != "intent.ledger":
                continue
            data = payload.get("data")
            if not isinstance(data, dict):
                continue
            rows.append(data)
        rows.sort(key=lambda d: _safe_float(d.get("ts"), 0.0))
        for data in rows:
            _apply(store, data)
    except Exception:
        return


def _apply(store: dict, payload: dict) -> dict:
    intent_id = str(payload.get("intent_id") or "").strip()
    if not intent_id:
        return {}

    intents = store.get("intents")
    if not isinstance(intents, dict):
        intents = {}
        store["intents"] = intents

    prev = intents.get(intent_id)
    rec = dict(prev or {})
    rec.update(payload)
    event_ts = _safe_float(payload.get("ts"), time.time())
    rec["intent_id"] = intent_id
    rec["ts"] = event_ts
    rec["stage"] = str(rec.get("stage") or "").upper().strip()
    rec["symbol"] = str(rec.get("symbol") or "").upper().strip()
    rec["side"] = str(rec.get("side") or "").lower().strip()
    rec["type"] = str(rec.get("type") or "").upper().strip()
    rec["status"] = str(rec.get("status") or "").lower().strip()
    rec["reason"] = str(rec.get("reason") or "")
    rec["is_exit"] = bool(rec.get("is_exit", False))
    rec["client_order_id"] = str(rec.get("client_order_id") or "").strip()
    rec["order_id"] = str(rec.get("order_id") or "").strip()
    rec["meta"] = dict(rec.get("meta") or {})
    created_prev = _safe_float((prev or {}).get("created_ts"), 0.0)
    rec["created_ts"] = created_prev if created_prev > 0 else event_ts
    unknown_prev = _safe_float((prev or {}).get("unknown_since_ts"), 0.0)
    stage_now = str(rec.get("stage") or "").upper().strip()
    if stage_now in _UNKNOWN_STAGES:
        rec["unknown_since_ts"] = (unknown_prev if unknown_prev > 0 else event_ts)
    elif stage_now == "DONE" and unknown_prev > 0:
        rec["unknown_resolve_sec"] = max(0.0, event_ts - unknown_prev)
        rec["unknown_since_ts"] = 0.0
    else:
        rec["unknown_since_ts"] = unknown_prev
    intents[intent_id] = rec

    by_order = store.get("by_order_id")
    if not isinstance(by_order, dict):
        by_order = {}
        store["by_order_id"] = by_order
    by_coid = store.get("by_client_order_id")
    if not isinstance(by_coid, dict):
        by_coid = {}
        store["by_client_order_id"] = by_coid

    oid = rec.get("order_id")
    if isinstance(oid, str) and oid:
        by_order[oid] = intent_id
    coid = rec.get("client_order_id")
    if isinstance(coid, str) and coid:
        by_coid[coid] = intent_id
    return rec


def _append(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, ensure_ascii=True) + "\n")


def record(
    bot,
    *,
    intent_id: str,
    stage: str,
    symbol: str = "",
    side: str = "",
    order_type: str = "",
    is_exit: bool = False,
    client_order_id: str = "",
    order_id: str = "",
    status: str = "",
    reason: str = "",
    meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    store = _ensure_store(bot)
    if store is None:
        return {}
    payload = {
        "ts": time.time(),
        "intent_id": str(intent_id or "").strip(),
        "stage": str(stage or "").upper().strip(),
        "symbol": str(symbol or "").upper().strip(),
        "side": str(side or "").lower().strip(),
        "type": str(order_type or "").upper().strip(),
        "is_exit": bool(is_exit),
        "client_order_id": str(client_order_id or "").strip(),
        "order_id": str(order_id or "").strip(),
        "status": str(status or "").lower().strip(),
        "reason": str(reason or ""),
        "meta": dict(meta or {}),
    }
    if not payload["intent_id"]:
        return {}
    rec = _apply(store, payload)
    try:
        _append(Path(str(store.get("path") or "logs/intent_ledger.jsonl")), payload)
    except Exception:
        pass
    if _event_journal is not None:
        try:
            _event_journal.append_event(bot, "intent.ledger", payload)
        except Exception:
            pass
    return dict(rec)


def get_intent(bot, intent_id: str) -> Optional[Dict[str, Any]]:
    store = _ensure_store(bot)
    if store is None:
        return None
    rec = (store.get("intents") or {}).get(str(intent_id or "").strip())
    return dict(rec) if isinstance(rec, dict) else None


def resolve_intent_id(bot, *, intent_id: str = "", client_order_id: str = "", order_id: str = "") -> str:
    iid = str(intent_id or "").strip()
    if iid:
        return iid
    store = _ensure_store(bot)
    if store is None:
        return ""
    if order_id:
        got = str((store.get("by_order_id") or {}).get(str(order_id or "").strip()) or "").strip()
        if got:
            return got
    if client_order_id:
        got = str((store.get("by_client_order_id") or {}).get(str(client_order_id or "").strip()) or "").strip()
        if got:
            return got
    return ""


def find_reusable_intent(
    bot,
    *,
    intent_id: str = "",
    client_order_id: str = "",
    order_id: str = "",
    max_age_sec: Optional[float] = None,
) -> Optional[Dict[str, Any]]:
    store = _ensure_store(bot)
    if store is None:
        return None
    iid = resolve_intent_id(bot, intent_id=intent_id, client_order_id=client_order_id, order_id=order_id)
    if not iid:
        return None
    rec = (store.get("intents") or {}).get(iid)
    if not isinstance(rec, dict):
        return None
    stage = str(rec.get("stage") or "").upper().strip()
    if stage not in _REUSABLE_STAGES:
        return None
    ttl = _reuse_max_age_sec(bot) if max_age_sec is None else max(0.0, _safe_float(max_age_sec, 0.0))
    if ttl > 0:
        age = max(0.0, time.time() - _safe_float(rec.get("ts"), 0.0))
        if age > ttl:
            return None
    return dict(rec)


def find_pending_unknown_intent(
    bot,
    *,
    intent_id: str = "",
    client_order_id: str = "",
    order_id: str = "",
    max_age_sec: Optional[float] = None,
) -> Optional[Dict[str, Any]]:
    store = _ensure_store(bot)
    if store is None:
        return None
    iid = resolve_intent_id(bot, intent_id=intent_id, client_order_id=client_order_id, order_id=order_id)
    if not iid:
        return None
    rec = (store.get("intents") or {}).get(iid)
    if not isinstance(rec, dict):
        return None
    stage = str(rec.get("stage") or "").upper().strip()
    if stage not in _UNKNOWN_STAGES:
        return None
    ttl = _unknown_max_age_sec(bot) if max_age_sec is None else max(0.0, _safe_float(max_age_sec, 0.0))
    if ttl > 0:
        age = max(0.0, time.time() - _safe_float(rec.get("ts"), 0.0))
        if age > ttl:
            return None
    return dict(rec)


def summary(bot, *, now_ts: Optional[float] = None) -> Dict[str, Any]:
    store = _ensure_store(bot)
    if store is None:
        return {
            "intents_total": 0,
            "intents_done": 0,
            "intent_unknown_count": 0,
            "intent_unknown_oldest_sec": 0.0,
            "intent_unknown_mean_resolve_sec": 0.0,
        }
    intents = store.get("intents")
    if not isinstance(intents, dict):
        intents = {}
    now = float(now_ts if now_ts is not None else time.time())
    unknown_count = 0
    unknown_oldest_sec = 0.0
    done_count = 0
    resolve_values: list[float] = []
    for rec_any in intents.values():
        rec = rec_any if isinstance(rec_any, dict) else {}
        stage = str(rec.get("stage") or "").upper().strip()
        if stage == "DONE":
            done_count += 1
        if stage in _UNKNOWN_STAGES:
            unknown_count += 1
            unknown_since = _safe_float(rec.get("unknown_since_ts"), 0.0)
            if unknown_since > 0:
                age = max(0.0, now - unknown_since)
                if age > unknown_oldest_sec:
                    unknown_oldest_sec = age
        resolve_sec = _safe_float(rec.get("unknown_resolve_sec"), 0.0)
        if resolve_sec > 0:
            resolve_values.append(resolve_sec)
    mean_resolve = (sum(resolve_values) / len(resolve_values)) if resolve_values else 0.0
    return {
        "intents_total": int(len(intents)),
        "intents_done": int(done_count),
        "intent_unknown_count": int(unknown_count),
        "intent_unknown_oldest_sec": float(unknown_oldest_sec),
        "intent_unknown_mean_resolve_sec": float(mean_resolve),
    }
