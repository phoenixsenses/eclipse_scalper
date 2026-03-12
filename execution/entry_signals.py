# execution/entry_signals.py — Signal parsing and loading helpers (pure functions)
# Extracted from entry_loop.py (Phase 7.3) — guardian-safe, never raises

from __future__ import annotations

from typing import Any, Callable, Dict, Optional

try:
    from core.micro_signal import PocketFilter  # type: ignore
except Exception:
    PocketFilter = None  # type: ignore


def _load_signal_fn() -> Optional[Callable]:
    """
    Locate a signal function without hard dependency.
    Preferred: strategies.eclipse_scalper.scalper_signal
    """
    try:
        from strategies.eclipse_scalper import scalper_signal as fn  # type: ignore
        if callable(fn):
            return fn
    except Exception:
        pass
    return None


def _parse_micro_pockets(raw: str) -> list:
    out = []
    if not callable(PocketFilter):
        return out
    for chunk in str(raw or "").split(";"):
        part = str(chunk or "").strip()
        if not part:
            continue
        vals = [x.strip() for x in part.split(",")]
        if len(vals) < 3:
            continue
        try:
            out.append(
                PocketFilter(
                    min_imbalance=float(vals[0]),
                    min_intensity=float(vals[1]),
                    max_spread=float(vals[2]),
                    priority=len(out),
                )
            )
        except Exception:
            continue
    return out


def _micro_signal_to_entry_sig(msig) -> Optional[Dict[str, Any]]:
    if msig is None:
        return None
    try:
        if hasattr(msig, "present") and hasattr(msig, "reason"):
            if not bool(getattr(msig, "present", False)):
                return None
            meta = dict(getattr(msig, "meta", {}) or {})
            inner = meta.get("signal")
            if inner is not None:
                msig = inner
        side = str(getattr(msig, "side", "")).strip().lower()
        if side not in ("buy", "sell"):
            return None
        feat = getattr(msig, "features", None)
        mark_px = float(getattr(feat, "mark_price", 0.0) or 0.0)
        otype = str(getattr(msig, "order_type", "limit") or "limit").strip().lower()
        sig: Dict[str, Any] = {
            "action": side,
            "confidence": float(getattr(msig, "confidence", 0.0) or 0.0),
            "type": ("limit" if otype == "limit" else "market"),
            "price": (mark_px if otype == "limit" and mark_px > 0 else None),
            "symbol": str(getattr(msig, "symbol", "") or ""),
            "pocket_name": str(getattr(msig, "pocket_name", "") or ""),
            "min_imbalance": float(getattr(feat, "imbalance", 0.0) or 0.0),
            "min_trade_intensity": float(getattr(feat, "trade_intensity", 0.0) or 0.0),
            "max_spread": float(getattr(feat, "spread", 0.0) or 0.0),
            "source": "micro_signal",
            "regime": str(getattr(msig, "regime", "UNKNOWN") or "UNKNOWN"),
            "regime_age_sec": float(getattr(msig, "regime_age_sec", 0.0) or 0.0),
            "fill_timeout_sec": float(getattr(msig, "fill_timeout_sec", 10.0) or 10.0),
        }
        return sig
    except Exception:
        return None


def _parse_action(sig: Dict[str, Any]) -> Optional[str]:
    a = str(sig.get("action") or sig.get("side") or "").strip().lower()
    if a in ("long", "buy"):
        return "buy"
    if a in ("short", "sell"):
        return "sell"
    return None


def _parse_order_type(sig: Dict[str, Any]) -> str:
    t = str(sig.get("type") or sig.get("order_type") or "market").strip().lower()
    return "limit" if t in ("limit",) else "market"


def _parse_amount(sig: Dict[str, Any]) -> Optional[float]:
    for key in ("amount", "qty", "size"):
        if key in sig:
            try:
                v = float(sig.get(key))
                return v if v > 0 else None
            except Exception:
                return None
    return None


def _parse_price(sig: Dict[str, Any]) -> Optional[float]:
    for key in ("price", "limit_price"):
        if key in sig:
            try:
                v = float(sig.get(key))
                return v if v > 0 else None
            except Exception:
                return None
    return None


def _order_filled(order: dict) -> float:
    try:
        if order is None:
            return 0.0
        if "filled" in order:
            return float(order.get("filled") or 0.0)
        info = order.get("info") or {}
        return float(info.get("executedQty") or 0.0)
    except Exception:
        return 0.0
