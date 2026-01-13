import ccxt.async_support as ccxt
import asyncio
import os
from dotenv import load_dotenv

load_dotenv()

async def test_futures_key():
    ex = ccxt.binanceusdm({
        'apiKey': os.getenv('BINANCE_API_KEY'),
        'secret': os.getenv('BINANCE_API_SECRET'),
        'enableRateLimit': True,
    })
    try:
        markets = await ex.load_markets()
        print(f"SUCCESS! Loaded {len(markets)} perpetual markets.")
        balance = await ex.fetch_balance()
        print(f"USDT Balance: {balance['total'].get('USDT', 0)}")
    except Exception as e:
        print(f"FAILED: {e}")
    finally:
        await ex.close()

asyncio.run(test_futures_key())