import asyncio
from exchanges.binance import get_exchange
from data.cache import GodEmperorDataOracle

class DummyState:
    positions = {}
    blacklist = {}
    last_exit_time = {}

class DummyBot:
    def __init__(self):
        self.ex = get_exchange()
        self.data = GodEmperorDataOracle()
        self.active_symbols = {"BTC/USDT"}
        self._shutdown = asyncio.Event()
        self.state = DummyState()

async def main():
    bot = DummyBot()
    try:
        sym = "BTC/USDT"
        t_task = asyncio.create_task(bot.data.poll_ticker(bot, sym))
        o_task = asyncio.create_task(bot.data.poll_ohlcv(bot, sym, "1m", bot.data.ohlcv))

        await asyncio.sleep(15)

        df = bot.data.get_df(sym, "1m")
        print("df rows:", len(df))
        print("last close:", df["c"].iloc[-1] if len(df) else None)
        print("price cache:", bot.data.price.get(sym))
        print("funding cache:", bot.data.funding.get(sym), "trend:", bot.data.get_funding_trend(sym))

        bot._shutdown.set()
        await asyncio.sleep(1)
    finally:
        await bot.ex.close()

if __name__ == "__main__":
    asyncio.run(main())
