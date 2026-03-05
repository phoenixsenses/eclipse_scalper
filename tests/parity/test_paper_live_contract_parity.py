from __future__ import annotations

from src.microphys.execution.engine import ExecutionRequest, build_default_engines


def test_paper_live_contract_parity_required_fields() -> None:
    engines = build_default_engines()
    req = ExecutionRequest(
        symbol="ETHUSDT",
        side="sell",
        entry_price=2200.0,
        exit_price=2198.5,
        notional=50.0,
        fee_bps=0.5,
        slippage_bps=0.2,
        ts_ms=1700000001234,
        order_id="ord-2",
    )
    paper = engines["paper"].execute(req).to_dict()
    live = engines["live"].execute(req).to_dict()
    required = {
        "venue",
        "symbol",
        "side",
        "entry_price",
        "exit_price",
        "qty_notional",
        "gross_return",
        "net_return",
        "fee_cost",
        "slippage_cost",
        "ts_ms",
        "order_id",
    }
    assert required.issubset(set(paper.keys()))
    assert required.issubset(set(live.keys()))
    # Parity contract: same numeric semantics across adapters for same input.
    for k in ("gross_return", "net_return", "fee_cost", "slippage_cost"):
        assert abs(float(paper[k]) - float(live[k])) <= 1e-12

