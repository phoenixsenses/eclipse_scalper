from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, Literal, Protocol

Side = Literal["buy", "sell"]


def _to_float(v: Any, default: float = 0.0) -> float:
    try:
        x = float(v)
        if x != x:
            return float(default)
        return x
    except Exception:
        return float(default)


@dataclass(frozen=True)
class ExecutionRequest:
    symbol: str
    side: Side
    entry_price: float
    exit_price: float
    notional: float = 1.0
    fee_bps: float = 0.0
    slippage_bps: float = 0.0
    ts_ms: int = 0
    order_id: str = ""


@dataclass(frozen=True)
class ExecutionResult:
    venue: str
    symbol: str
    side: Side
    entry_price: float
    exit_price: float
    qty_notional: float
    gross_return: float
    net_return: float
    fee_cost: float
    slippage_cost: float
    ts_ms: int
    order_id: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ExecutionAdapter(Protocol):
    name: str

    def execute(self, req: ExecutionRequest) -> ExecutionResult:
        ...


class DeterministicAdapter:
    def __init__(self, name: str) -> None:
        self.name = str(name)

    def execute(self, req: ExecutionRequest) -> ExecutionResult:
        ep = max(1e-12, _to_float(req.entry_price, 0.0))
        xp = max(1e-12, _to_float(req.exit_price, 0.0))
        side = str(req.side).lower()
        gross = ((xp - ep) / ep) if side == "buy" else ((ep - xp) / ep)
        fee = max(0.0, _to_float(req.fee_bps, 0.0)) / 10000.0
        slip = max(0.0, _to_float(req.slippage_bps, 0.0)) / 10000.0
        net = gross - fee - slip
        return ExecutionResult(
            venue=self.name,
            symbol=str(req.symbol),
            side=("buy" if side == "buy" else "sell"),
            entry_price=float(ep),
            exit_price=float(xp),
            qty_notional=max(0.0, _to_float(req.notional, 0.0)),
            gross_return=float(gross),
            net_return=float(net),
            fee_cost=float(fee),
            slippage_cost=float(slip),
            ts_ms=int(req.ts_ms or 0),
            order_id=str(req.order_id or ""),
        )


class ExecutionEngine:
    """Unified execution interface for backtest/paper/live adapters."""

    def __init__(self, adapter: ExecutionAdapter) -> None:
        self.adapter = adapter

    def execute(self, req: ExecutionRequest) -> ExecutionResult:
        return self.adapter.execute(req)


def build_default_engines() -> Dict[str, ExecutionEngine]:
    return {
        "backtest": ExecutionEngine(DeterministicAdapter("backtest")),
        "paper": ExecutionEngine(DeterministicAdapter("paper")),
        "live": ExecutionEngine(DeterministicAdapter("live")),
    }

