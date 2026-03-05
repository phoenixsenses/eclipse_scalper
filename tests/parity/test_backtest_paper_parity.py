from __future__ import annotations

from src.microphys.execution.engine import ExecutionRequest, build_default_engines


def test_backtest_paper_parity_identical_inputs() -> None:
    engines = build_default_engines()
    req = ExecutionRequest(
        symbol="ETHUSDT",
        side="buy",
        entry_price=2100.0,
        exit_price=2101.2,
        notional=100.0,
        fee_bps=0.5,
        slippage_bps=0.1,
        ts_ms=1700000000000,
        order_id="ord-1",
    )
    bt = engines["backtest"].execute(req)
    pp = engines["paper"].execute(req)
    assert abs(bt.net_return - pp.net_return) <= 1e-12
    assert abs(bt.gross_return - pp.gross_return) <= 1e-12
    assert abs(bt.fee_cost - pp.fee_cost) <= 1e-12
    assert abs(bt.slippage_cost - pp.slippage_cost) <= 1e-12

