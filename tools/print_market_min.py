from dotenv import load_dotenv
load_dotenv()

import os, ccxt

ex = ccxt.binance({
    "apiKey": os.getenv("BINANCE_API_KEY"),
    "secret": os.getenv("BINANCE_API_SECRET"),
    "options": {"defaultType": "future"},
    "enableRateLimit": True,
})

ex.load_markets()

for sym in ["BNB/USDT:USDT", "DOGE/USDT:USDT"]:
    m = ex.market(sym)
    limits = m.get("limits") or {}
    min_amt = (limits.get("amount") or {}).get("min")
    min_cost = (limits.get("cost") or {}).get("min")
    print(sym, "min amount:", min_amt, "min cost:", min_cost)
