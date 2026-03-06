from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, Literal, Optional, Union

EVENT_SCHEMA_VERSION = 1

EventType = Literal["order_intent", "order_ack", "fill", "reject"]
SideType = Literal["BUY", "SELL"]
OrderType = Literal["LIMIT", "MARKET"]
TifType = Literal["GTC", "IOC", "FOK"]
AckStatus = Literal["ACKED", "CANCELED"]
LiquidityType = Literal["maker", "taker", "unknown"]


def _to_int(x: Any, default: int = 0) -> int:
    try:
        return int(float(x))
    except Exception:
        return int(default)


def _to_float(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
        if v != v:  # NaN
            return float(default)
        return v
    except Exception:
        return float(default)


def _norm_side(side: Any) -> str:
    s = str(side or "").strip().upper()
    if s in {"LONG", "BUY", "B"}:
        return "BUY"
    if s in {"SHORT", "SELL", "S"}:
        return "SELL"
    return s


def _norm_order_type(order_type: Any) -> str:
    s = str(order_type or "").strip().upper()
    return s if s in {"LIMIT", "MARKET"} else "LIMIT"


def _norm_tif(tif: Any) -> str:
    s = str(tif or "").strip().upper()
    return s if s in {"GTC", "IOC", "FOK"} else "GTC"


def _norm_liquidity(v: Any) -> str:
    s = str(v or "").strip().lower()
    if s in {"maker", "taker"}:
        return s
    return "unknown"


@dataclass(frozen=True)
class OrderIntent:
    event_type: Literal["order_intent"]
    schema_version: int
    event_id: str
    ts_ms: int
    source: str
    symbol: str
    order_id: str
    client_order_id: str
    side: SideType
    order_type: OrderType
    tif: TifType
    qty: float
    limit_price: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class OrderAck:
    event_type: Literal["order_ack"]
    schema_version: int
    event_id: str
    ts_ms: int
    source: str
    symbol: str
    order_id: str
    client_order_id: str
    side: SideType
    status: AckStatus
    ack_price: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class FillEvent:
    event_type: Literal["fill"]
    schema_version: int
    event_id: str
    ts_ms: int
    source: str
    symbol: str
    order_id: str
    client_order_id: str
    side: SideType
    fill_qty: float
    fill_price: float
    cumulative_qty: float
    remaining_qty: float
    liquidity: LiquidityType = "unknown"
    fee_bps: float = 0.0
    effective_cost_bps: float = 0.0
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RejectEvent:
    event_type: Literal["reject"]
    schema_version: int
    event_id: str
    ts_ms: int
    source: str
    symbol: str
    order_id: str
    client_order_id: str
    side: SideType
    reason_code: str
    reason_text: str
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


ExecutionEvent = Union[OrderIntent, OrderAck, FillEvent, RejectEvent]


def event_from_dict(raw: Dict[str, Any]) -> ExecutionEvent:
    d = dict(raw or {})
    et = str(d.get("event_type") or "").strip().lower()
    common = {
        "schema_version": _to_int(d.get("schema_version"), EVENT_SCHEMA_VERSION),
        "event_id": str(d.get("event_id") or ""),
        "ts_ms": _to_int(d.get("ts_ms"), 0),
        "source": str(d.get("source") or ""),
        "symbol": str(d.get("symbol") or ""),
        "order_id": str(d.get("order_id") or ""),
        "client_order_id": str(d.get("client_order_id") or ""),
        "side": _norm_side(d.get("side")),
        "metadata": d.get("metadata") if isinstance(d.get("metadata"), dict) else None,
    }
    if et == "order_intent":
        return OrderIntent(
            event_type="order_intent",
            **common,
            order_type=_norm_order_type(d.get("order_type")),
            tif=_norm_tif(d.get("tif")),
            qty=_to_float(d.get("qty"), 0.0),
            limit_price=(_to_float(d.get("limit_price"), 0.0) if d.get("limit_price") is not None else None),
        )
    if et == "order_ack":
        status = str(d.get("status") or "").strip().upper()
        if status not in {"ACKED", "CANCELED"}:
            status = "ACKED"
        return OrderAck(
            event_type="order_ack",
            **common,
            status=status,  # type: ignore[arg-type]
            ack_price=(_to_float(d.get("ack_price"), 0.0) if d.get("ack_price") is not None else None),
        )
    if et == "fill":
        return FillEvent(
            event_type="fill",
            **common,
            fill_qty=_to_float(d.get("fill_qty"), 0.0),
            fill_price=_to_float(d.get("fill_price"), 0.0),
            cumulative_qty=_to_float(d.get("cumulative_qty"), 0.0),
            remaining_qty=_to_float(d.get("remaining_qty"), 0.0),
            liquidity=_norm_liquidity(d.get("liquidity")),  # type: ignore[arg-type]
            fee_bps=_to_float(d.get("fee_bps"), 0.0),
            effective_cost_bps=_to_float(d.get("effective_cost_bps"), 0.0),
        )
    if et == "reject":
        return RejectEvent(
            event_type="reject",
            **common,
            reason_code=str(d.get("reason_code") or "UNKNOWN"),
            reason_text=str(d.get("reason_text") or ""),
        )
    raise ValueError(f"unknown event_type={et!r}")


def validate_event(ev: ExecutionEvent) -> list[str]:
    errs: list[str] = []
    if int(getattr(ev, "schema_version", 0) or 0) != EVENT_SCHEMA_VERSION:
        errs.append(f"schema_version_invalid:{getattr(ev, 'schema_version', None)}")
    if not str(getattr(ev, "event_id", "") or "").strip():
        errs.append("event_id_missing")
    if int(getattr(ev, "ts_ms", 0) or 0) <= 0:
        errs.append("ts_ms_non_positive")
    if not str(getattr(ev, "symbol", "") or "").strip():
        errs.append("symbol_missing")
    if not str(getattr(ev, "order_id", "") or "").strip():
        errs.append("order_id_missing")
    if not str(getattr(ev, "client_order_id", "") or "").strip():
        errs.append("client_order_id_missing")
    if str(getattr(ev, "side", "") or "") not in {"BUY", "SELL"}:
        errs.append(f"side_invalid:{getattr(ev, 'side', None)}")

    if isinstance(ev, OrderIntent):
        if float(ev.qty) <= 0:
            errs.append("intent_qty_non_positive")
        if str(ev.order_type) == "LIMIT":
            if ev.limit_price is None or float(ev.limit_price) <= 0:
                errs.append("intent_limit_price_non_positive")
    elif isinstance(ev, OrderAck):
        if str(ev.status) not in {"ACKED", "CANCELED"}:
            errs.append(f"ack_status_invalid:{ev.status}")
        if ev.ack_price is not None and float(ev.ack_price) <= 0:
            errs.append("ack_price_non_positive")
    elif isinstance(ev, FillEvent):
        if float(ev.fill_qty) <= 0:
            errs.append("fill_qty_non_positive")
        if float(ev.fill_price) <= 0:
            errs.append("fill_price_non_positive")
        if float(ev.cumulative_qty) < 0 or float(ev.remaining_qty) < 0:
            errs.append("fill_qty_negative_state")
        if float(ev.cumulative_qty) + 1e-12 < float(ev.fill_qty):
            errs.append("fill_cumulative_lt_fill")
        if not (-1000.0 <= float(ev.effective_cost_bps) <= 1000.0):
            errs.append(f"effective_cost_bps_out_of_bounds:{ev.effective_cost_bps}")
        if str(ev.liquidity) not in {"maker", "taker", "unknown"}:
            errs.append(f"liquidity_invalid:{ev.liquidity}")
    elif isinstance(ev, RejectEvent):
        if not str(ev.reason_code or "").strip():
            errs.append("reject_reason_code_missing")
    else:
        errs.append("event_unknown_type")
    return errs


def validate_event_sequence(events: Iterable[ExecutionEvent]) -> list[str]:
    errs: list[str] = []
    per_order_last_ts: Dict[str, int] = {}
    per_order_state: Dict[str, str] = {}
    terminal_states = {"FILLED", "REJECTED", "CANCELED"}

    for i, ev in enumerate(events):
        order_id = str(getattr(ev, "order_id", "") or "")
        ts_ms = int(getattr(ev, "ts_ms", 0) or 0)
        if not order_id:
            errs.append(f"seq[{i}]:order_id_missing")
            continue
        prev_ts = per_order_last_ts.get(order_id)
        if prev_ts is not None and ts_ms < prev_ts:
            errs.append(f"seq[{i}]:ts_ms_not_monotonic:{order_id}:{ts_ms}<{prev_ts}")
        per_order_last_ts[order_id] = ts_ms

        st = per_order_state.get(order_id, "NONE")
        if st in terminal_states:
            errs.append(f"seq[{i}]:event_after_terminal:{order_id}:{st}")
            continue
        if isinstance(ev, OrderIntent):
            if st != "NONE":
                errs.append(f"seq[{i}]:intent_after_{st}:{order_id}")
            per_order_state[order_id] = "INTENT"
        elif isinstance(ev, OrderAck):
            if st not in {"INTENT", "ACKED", "PARTIAL"}:
                errs.append(f"seq[{i}]:ack_without_intent:{order_id}:{st}")
            per_order_state[order_id] = str(ev.status)
        elif isinstance(ev, FillEvent):
            if st not in {"INTENT", "ACKED", "PARTIAL"}:
                errs.append(f"seq[{i}]:fill_without_intent_or_ack:{order_id}:{st}")
            if float(ev.remaining_qty) <= 0:
                per_order_state[order_id] = "FILLED"
            else:
                per_order_state[order_id] = "PARTIAL"
        elif isinstance(ev, RejectEvent):
            if st not in {"NONE", "INTENT", "ACKED"}:
                errs.append(f"seq[{i}]:reject_after_{st}:{order_id}")
            per_order_state[order_id] = "REJECTED"
    return errs

