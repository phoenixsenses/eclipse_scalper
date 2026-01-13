import asyncio
from exchanges.binance import get_exchange

async def main():
    ex = get_exchange()
    try:
        mkts = await ex.fetch_markets()
        print("markets:", len(mkts))

        t = await ex.fetch_ticker("BTC/USDT")
        print("ticker last:", t.get("last"))

        fr = await ex.fetch_funding_rate("BTC/USDT")
        print("funding:", fr)

        # This should be BLOCKED in dry run:
        o = await ex.create_order(
            symbol="BTC/USDT",
            type="market",
            side="buy",
            amount=0.001,
            params={}
        )
        print("order:", o)
    finally:
        await ex.close()

if __name__ == "__main__":
    asyncio.run(main())
