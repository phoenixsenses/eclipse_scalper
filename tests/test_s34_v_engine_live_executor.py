import asyncio

from tools.s34_v_engine_live_executor import (
    RULE_NAME,
    active_entry_ref,
    allowed_rules,
    client_id,
    has_s34ve_protective_stop,
    has_active_order,
    is_s34ve_order,
    position_amount,
    reconcile_exchange_safety,
    stop_price_long,
)


def test_allowed_rules_parses_comma_list():
    env = {"S34_LIVE_ALLOWED_RULES": f"{RULE_NAME},OTHER"}

    assert allowed_rules(env) == {RULE_NAME, "OTHER"}


def test_client_id_is_binance_sized_and_stable():
    cid = client_id("S34_V_ENGINE:12345678901234567890", "E1")

    assert cid.startswith("S34VE")
    assert len(cid) <= 32
    assert cid == client_id("S34_V_ENGINE:12345678901234567890", "E1")


def test_has_active_order_only_for_live_lifecycle_states():
    assert not has_active_order({"active": None})
    assert has_active_order({"active": {"status": "ENTRY_INITIAL_OPEN"}})
    assert has_active_order({"active": {"status": "POSITION_OPEN"}})
    assert not has_active_order({"active": {"status": "CLOSED"}})


def test_position_amount_reads_binance_info_position_side():
    positions = [
        {"symbol": "ETH/USDT:USDT", "info": {"positionSide": "LONG", "positionAmt": "0.25"}},
        {"symbol": "ETH/USDT:USDT", "info": {"positionSide": "SHORT", "positionAmt": "-0.10"}},
    ]

    assert position_amount(positions, symbol="ETHUSDT", direction="LONG") == 0.25
    assert position_amount(positions, symbol="ETHUSDT", direction="SHORT") == 0.10


def test_stop_price_long():
    assert stop_price_long(100.0, 150.0) == 98.5


def test_active_entry_ref_prefers_replacement_when_replaced():
    active = {
        "status": "ENTRY_REPLACED_OPEN",
        "initial_order": {"limit_price": 100.0},
        "replace_order": {"limit_price": 99.0},
    }

    assert active_entry_ref(active) == 99.0
    assert active_entry_ref({**active, "status": "ENTRY_INITIAL_OPEN"}) == 100.0


def test_s34ve_order_and_protective_stop_detection():
    entry = {"id": "1", "info": {"clientOrderId": client_id("event", "E1"), "type": "LIMIT"}}
    stop = {
        "id": "2",
        "info": {"clientOrderId": client_id("event", "SL"), "type": "STOP_MARKET", "reduceOnly": "true"},
    }

    assert is_s34ve_order(entry)
    assert has_s34ve_protective_stop([entry, stop])


class FakeExchange:
    def __init__(self, *, open_orders=None, positions=None):
        self.open_orders = open_orders or []
        self.positions = positions or []
        self.cancelled = []
        self.created = []

    async def fetch_open_orders(self, symbol):
        return self.open_orders

    async def fetch_positions(self, symbols=None):
        return self.positions

    async def cancel_order(self, order_id, symbol):
        self.cancelled.append((order_id, symbol))
        return {"id": order_id, "status": "canceled"}

    async def fetch_ticker(self, symbol):
        return {"last": 100.0}

    async def fetch_markets(self):
        return []

    def amount_to_precision(self, symbol, amount):
        return str(amount)

    def price_to_precision(self, symbol, price):
        return str(price)

    async def create_order(self, symbol, typ, side, amount, price=None, params=None):
        order = {"symbol": symbol, "type": typ, "side": side, "amount": amount, "price": price, "params": params or {}}
        self.created.append(order)
        return order


def test_reconcile_cancels_stale_s34ve_orders_before_new_entry():
    stale = {"id": "123", "info": {"clientOrderId": client_id("old", "E1"), "type": "LIMIT"}}
    ex = FakeExchange(open_orders=[stale], positions=[])
    state = {"processed_event_ids": {}, "active": None, "orders": [], "status": {}}

    result = asyncio.run(reconcile_exchange_safety({}, state, ex, dry_run=False))

    assert result["allow_new_entry"] is False
    assert result["reason"] == "cancelled_stale_s34ve_orders"
    assert ex.cancelled == [("123", "ETHUSDT")]
    assert state["orders"][-1]["action"] == "cancel_stale_s34ve_orders"


def test_reconcile_orphan_position_places_emergency_stop():
    ex = FakeExchange(
        positions=[{"symbol": "ETH/USDT:USDT", "info": {"positionSide": "LONG", "positionAmt": "0.25"}}]
    )
    state = {"processed_event_ids": {}, "active": None, "orders": [], "status": {}}

    result = asyncio.run(reconcile_exchange_safety({}, state, ex, dry_run=False))

    assert result["allow_new_entry"] is False
    assert result["reason"] == "orphan_position_reconciled"
    assert state["active"]["status"] == "POSITION_OPEN"
    assert state["active"]["orphan_reconciled"] is True
    assert ex.created[-1]["type"] == "STOP_MARKET"
    assert ex.created[-1]["side"] == "sell"
    assert ex.created[-1]["params"]["reduceOnly"] is True
